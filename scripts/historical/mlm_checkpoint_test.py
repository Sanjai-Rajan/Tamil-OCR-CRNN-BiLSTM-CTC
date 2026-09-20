import os
import shutil
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "indic_bert"
TEST_OUT_DIR = PROJECT_ROOT / "outputs" / "checkpoint_test"

def generate_dummy_dataset(tokenizer, num_samples=100):
    texts = ["இது ஒரு தமிழ் சோதனை ஆகும்." for _ in range(num_samples)]
    encodings = tokenizer(texts, padding="max_length", truncation=True, max_length=128)
    return Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask']
    })

def main():
    print("=== PHASE 2: CHECKPOINT RESUME TEST ===")
    
    if TEST_OUT_DIR.exists():
        shutil.rmtree(TEST_OUT_DIR)
    
    print("Loading tokenizer and model locally...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(str(MODEL_DIR), local_files_only=True)
    
    dummy_dataset = generate_dummy_dataset(tokenizer, 200)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    
    training_args = TrainingArguments(
        output_dir=str(TEST_OUT_DIR),
        max_steps=10,
        per_device_train_batch_size=2,
        save_strategy="steps",
        save_steps=10,
        logging_steps=5,
        learning_rate=2e-5,
        report_to="none",
        use_cpu=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dummy_dataset,
        data_collator=data_collator,
    )
    
    print("\n--- Running initial training (10 steps) ---")
    trainer.train()
    
    checkpoint_dir = TEST_OUT_DIR / "checkpoint-10"
    if checkpoint_dir.exists():
        print("CHECKPOINT_SAVE = PASS")
    else:
        print("CHECKPOINT_SAVE = FAIL")
        return
        
    print("\n--- Reloading and Resuming (up to 20 steps) ---")
    
    model_reloaded = AutoModelForMaskedLM.from_pretrained(str(MODEL_DIR), local_files_only=True)
    
    training_args_resume = TrainingArguments(
        output_dir=str(TEST_OUT_DIR),
        max_steps=20,
        per_device_train_batch_size=2,
        save_strategy="steps",
        save_steps=10,
        logging_steps=5,
        learning_rate=2e-5,
        report_to="none",
        use_cpu=False
    )
    
    trainer_resume = Trainer(
        model=model_reloaded,
        args=training_args_resume,
        train_dataset=dummy_dataset,
        data_collator=data_collator,
    )
    
    try:
        train_result = trainer_resume.train(resume_from_checkpoint=str(checkpoint_dir))
        print("CHECKPOINT_RELOAD = PASS")
    except Exception as e:
        print(f"CHECKPOINT_RELOAD = FAIL (Error: {e})")
        return
        
    state = trainer_resume.state
    if state.global_step == 20:
        print("GLOBAL_STEP_RESUME = PASS")
    else:
        print(f"GLOBAL_STEP_RESUME = FAIL (Ended at {state.global_step})")
        
    ckpt_20 = TEST_OUT_DIR / "checkpoint-20"
    opt_exists = (ckpt_20 / "optimizer.pt").exists()
    sched_exists = (ckpt_20 / "scheduler.pt").exists()
    
    if opt_exists:
        print("OPTIMIZER_STATE = PASS")
    else:
        print("OPTIMIZER_STATE = FAIL")
        
    if sched_exists:
        print("SCHEDULER_STATE = PASS")
    else:
        print("SCHEDULER_STATE = FAIL")
        
    shutil.rmtree(TEST_OUT_DIR)
    print("Test checkpoint cleaned up successfully.")

if __name__ == "__main__":
    main()
