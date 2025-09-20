from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model
import torch
import os
import json
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login

# Disable bitsandbytes warnings and errors before any imports
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["DISABLE_BITSANDBYTES_WARN"] = "1"
# Force single GPU mode and disable accelerate auto-optimization
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["ACCELERATE_USE_CPU"] = "false"
os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
os.environ["ACCELERATE_USE_FSDP"] = "false"

# Load environment variables
load_dotenv()

# Add HuggingFace login for accessing gated models
login(token=os.getenv('HUGGINGFACE_TOKEN'))

# Completely disable bitsandbytes detection by PEFT
import sys
import importlib.util

# Mock bitsandbytes module that always returns None for find_spec
class MockBitsandbytesModule:
    """Mock module to replace bitsandbytes and prevent PEFT from detecting it"""
    def __getattr__(self, name):
        return None

# Replace bitsandbytes in sys.modules
sys.modules['bitsandbytes'] = MockBitsandbytesModule()

# Override importlib.util.find_spec for bitsandbytes
original_find_spec = importlib.util.find_spec

def patched_find_spec(name, package=None):
    if name == 'bitsandbytes':
        return None  # Always return None for bitsandbytes
    return original_find_spec(name, package)

importlib.util.find_spec = patched_find_spec

def setup_logging(log_level=logging.INFO):
    """Configure logging with separate output and error files (overwrite mode)."""
    log_dir = Path("/home3/tgpp65/Final_Project/final_test/logs")
    log_dir.mkdir(exist_ok=True)
    
    # Define log files
    out_file = log_dir / "fine_tune.out"
    err_file = log_dir / "fine_tune.err"
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # Create file handler for general output (INFO and above) - OVERWRITE mode
    out_handler = logging.FileHandler(out_file, mode='w')  # 'w' = overwrite
    out_handler.setLevel(logging.INFO)
    out_handler.setFormatter(file_formatter)
    
    # Create file handler for errors only (WARNING and above) - OVERWRITE mode
    err_handler = logging.FileHandler(err_file, mode='w')  # 'w' = overwrite
    err_handler.setLevel(logging.WARNING)
    err_handler.setFormatter(file_formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=[out_handler, err_handler, console_handler]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Output: {out_file}, Errors: {err_file}")
    logger.info("Log files will be overwritten on each run")
    
    return logger

# Initialize logger
logger = setup_logging()

def load_text_files(directory):
    """Load text content from markdown files."""
    texts = []
    logger.info(f"Looking for markdown files in directory: {directory}")
    
    if not os.path.exists(directory):
        logger.error(f"Directory '{directory}' does not exist")
        raise FileNotFoundError(f"Directory '{directory}' does not exist")
    
    files = [f for f in os.listdir(directory) if f.endswith('.md')]
    logger.info(f"Found {len(files)} markdown files")
    
    for file in files:
        file_path = os.path.join(directory, file)
        logger.info(f"Processing file: {file}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                texts.append(content)
                logger.info(f"Successfully loaded content from {file}")
        except Exception as e:
            logger.error(f"Unexpected error processing {file}: {e}")
    
    logger.info(f"Total number of texts extracted: {len(texts)}")
    if len(texts) == 0:
        logger.error("No texts were extracted from the markdown files")
        raise ValueError("No texts were extracted from the markdown files")
    
    return texts

def prepare_dataset(texts, tokenizer):
    """Tokenize texts and prepare them for training."""
    logger.info("Tokenizing texts for training")
    
    # Tokenize each text individually to avoid issues
    all_input_ids = []
    all_attention_masks = []
    
    for i, text in enumerate(texts):
        logger.debug(f"Tokenizing text {i+1}/{len(texts)}")
        encoding = tokenizer(
            text, 
            truncation=True, 
            padding=True, 
            max_length=512,
            return_tensors="pt"
        )
        all_input_ids.append(encoding.input_ids.squeeze())
        all_attention_masks.append(encoding.attention_mask.squeeze())
    
    # Create dataset
    dataset = Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_input_ids  # For causal LM, labels are the same as input_ids
    })
    
    logger.info(f"Dataset created with {len(dataset)} examples")
    return dataset

def load_json_files(directory):
    """Load improved content from JSON files in the specified directory."""
    texts = []
    logger.info(f"Looking for JSON files in directory: {directory}")
    
    if not os.path.exists(directory):
        logger.error(f"Directory '{directory}' does not exist")
        raise FileNotFoundError(f"Directory '{directory}' does not exist")
    
    files = [f for f in os.listdir(directory) if f.endswith('.json')]
    logger.info(f"Found {len(files)} JSON files")
    
    # These are the sections we want to extract, in order
    sections = ['INTRODUCTION', 'METHODOLOGY', 'RESULTS', 'DISCUSSION', 'CONCLUSION']
    
    for file in files:
        file_path = os.path.join(directory, file)
        logger.info(f"Processing file: {file}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Check if 'sections' key exists
                if 'sections' in data:
                    # Create a prompt template for each section's improved content
                    for section in sections:
                        if section in data['sections']:
                            section_data = data['sections'][section]
                            # Extract only the 'improved' content if available
                            if isinstance(section_data, dict) and 'improved' in section_data:
                                improved_text = section_data['improved']
                                # Create a formatted text with section name and content
                                formatted_text = f"### {section}:\n{improved_text}\n\n"
                                texts.append(formatted_text)
                                logger.info(f"Successfully extracted improved {section} from {file}")
                else:
                    logger.warning(f"No 'sections' field found in {file}")
                    logger.info(f"Available keys in {file}: {list(data.keys())}")
                    
        except json.JSONDecodeError as e:
            logger.error(f"Error reading JSON from {file}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error processing {file}: {e}")
    
    logger.info(f"Total number of texts extracted: {len(texts)}")
    if len(texts) == 0:
        logger.error("No texts were extracted from the JSON files")
        raise ValueError("No texts were extracted from the JSON files")
    
    return texts

def main():
    
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"  # Recommended for academic fine-tuning
    
    # Define directory for markdown files
    directory = 'chapter_markdowns'
    
    logger.info(f"Starting fine-tuning using content from markdown files in {directory}")
    
    # Load model and tokenizer (without quantization for better compatibility)
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # Configure LoRA BEFORE moving to GPU
    logger.info("Configuring LoRA settings")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Create PEFT model BEFORE moving to GPU
    logger.info("Creating PEFT model")
    model = get_peft_model(model, lora_config)
    
    # NOW move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info(f"Model moved to GPU: {torch.cuda.get_device_name()}")
    
    # Ensure model is in training mode
    model.train()
    
    # Enable gradients for all parameters (especially LoRA parameters)
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.debug(f"Trainable parameter: {name}")
        else:
            logger.debug(f"Frozen parameter: {name}")
    
    # Print trainable parameters info
    model.print_trainable_parameters()
    
    # Verify that some parameters require gradients
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    if trainable_params == 0:
        logger.error("No trainable parameters found! This will cause the training error.")
        raise RuntimeError("No trainable parameters found!")
    
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load texts from markdown files
    logger.info("Loading training texts from markdown files")
    texts = load_text_files(directory)
    
    # Prepare dataset
    logger.info("Preparing dataset for training")
    dataset = prepare_dataset(texts, tokenizer)
    
    # Define training arguments (optimized for stability and avoiding accelerate issues)
    logger.info("Setting up training arguments")
    training_args = TrainingArguments(
        output_dir="./fine_tuned_model",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        save_steps=50,
        save_total_limit=2,
        learning_rate=2e-4,
        warmup_steps=50,
        logging_dir='./logs',
        logging_steps=10,
        fp16=False,  # Disable fp16 to avoid precision issues
        gradient_checkpointing=False,  # Disable gradient checkpointing
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        # Additional stability settings
        max_grad_norm=1.0,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        # Disable problematic features
        dataloader_num_workers=0,  # Disable multiprocessing
        disable_tqdm=False,
        load_best_model_at_end=False,
        metric_for_best_model=None,
        greater_is_better=None,
        # Force single device training
        no_cuda=False,
        local_rank=-1,
        ddp_find_unused_parameters=False,
    )
    
    # Create data collator
    logger.info("Creating data collator")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer with explicit model preparation
    logger.info("Initializing trainer")
    
    # Ensure model is properly prepared for training
    model.train()
    
    # Additional check: manually verify gradients work
    dummy_input = torch.ones(1, 10, dtype=torch.long)
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
    
    try:
        with torch.cuda.amp.autocast():
            dummy_output = model(dummy_input)
            if hasattr(dummy_output, 'logits'):
                dummy_loss = dummy_output.logits.mean()
            else:
                dummy_loss = dummy_output.mean()
        dummy_loss.backward()
        logger.info("Gradient test passed - model is ready for training")
    except Exception as e:
        logger.error(f"Gradient test failed: {e}")
        raise RuntimeError(f"Model gradient computation failed: {e}")
    
    # Clear the dummy gradients
    model.zero_grad()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Start training
    logger.info("Starting model training")
    
    try:
        trainer.train()
    except RuntimeError as e:
        if "does not require grad" in str(e):
            logger.warning("Trainer failed with gradient issue, trying manual training loop")
            
            # Manual training loop as fallback
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
            
            for epoch in range(1):
                total_loss = 0
                for step, batch in enumerate(trainer.get_train_dataloader()):
                    optimizer.zero_grad()
                    
                    # Move batch to device
                    if torch.cuda.is_available():
                        batch = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = model(**batch)
                    loss = outputs.loss
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                    if step % 10 == 0:
                        logger.info(f"Step {step}, Loss: {loss.item():.4f}")
                    
                    if step >= 100:  # Limit steps for testing
                        break
                
                logger.info(f"Epoch {epoch}, Average Loss: {total_loss/(step+1):.4f}")
        else:
            raise e
    
    # Save the fine-tuned model
    logger.info("Saving fine-tuned model")
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    
    logger.info("Completed fine-tuning using markdown content")
    logger.info("Fine-tuning process completed successfully!")

if __name__ == "__main__":
    try:
        logger.info("=== Starting Fine-Tuning Process ===")
        main()
        logger.info("=== Fine-Tuning Process Completed Successfully ===")
    except KeyboardInterrupt:
        logger.warning("Fine-tuning process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Fatal error in fine-tuning process: {e}", exc_info=True)
        sys.exit(1)

