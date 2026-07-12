# src/train.py
import os
import sys
import logging
import torch
from transformers import (
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    DataCollatorForSeq2Seq,
    TrainerCallback,
    BitsAndBytesConfig,
    utils as hf_utils
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import config
from dataset import get_tokenized_dataset

logger = logging.getLogger(__name__)

def setup_logging():
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(name)s) - %(message)s",
        handlers=[
            logging.FileHandler(config.LOG_FILE_PATH, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    hf_utils.logging.set_verbosity_info()
    hf_utils.logging.enable_default_handler()
    hf_utils.logging.enable_explicit_format()

class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logger.info(f"Step Metrics: {logs}")

def main():
    setup_logging()
    
    logger.info("Initializing Data Preparation stage...")
    try:
        dataset, tokenizer = get_tokenized_dataset()
        logger.info(f"Dataset ready. Processing tokens up to length {config.MAX_INPUT_LEN}.")
    except Exception as e:
        logger.error(f"Failed during dataset initialization: {str(e)}")
        sys.exit(1)
    
    logger.info(f"Loading base model in compressed 4-bit space: {config.MODEL_NAME}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,       
        low_cpu_mem_usage=True           
    )
    
    # Prepares base layers for quantization
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]  
    )
    
    model = get_peft_model(model, peft_config)
    
    # ------------------------------------------------------------------
    # FIX: DEEP ARCHITECTURE GRADIENT HOOK
    # This manually forces gradient tracking on all internal embedding 
    # sub-modules across both the Encoder and the Decoder structures.
    # ------------------------------------------------------------------
    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)

    hook_count = 0
    for name, module in model.named_modules():
        if "embed_tokens" in name or "shared" in name:
            module.register_forward_hook(make_inputs_require_grad)
            hook_count += 1
    logger.info(f"Successfully secured {hook_count} internal embedding gradient chains.")
    # ------------------------------------------------------------------

    logger.info("LoRA Layers injected successfully.")
    model.print_trainable_parameters()
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.OUTPUT_DIR,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUMULATION,
        learning_rate=config.LEARNING_RATE,
        num_train_epochs=config.EPOCHS,
        predict_with_generate=True,
        logging_dir=config.LOGS_DIR,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,                             
        gradient_checkpointing=True, # Let the trainer securely orchestrate checkpointing
        gradient_checkpointing_kwargs={"use_reentrant": False},  
        optim="paged_adamw_8bit",              
        report_to="none"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[LoggingCallback()]
    )

    logger.info("--- Initiating QLoRA Model Fine-Tuning Optimization ---")
    trainer.train()
    
    logger.info("--- Training Complete. Saving LoRA Adapters... ---")
    trainer.save_model(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)
    logger.info(f"Fine-tuned adapter pipeline saved successfully to '{config.OUTPUT_DIR}'")

if __name__ == "__main__":
    main()