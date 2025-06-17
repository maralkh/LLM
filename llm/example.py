# example.py - نمونه استفاده از training infrastructure

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from training_infra import TrainingConfig, Trainer, EarlyStopping

# مثال 1: Simple Classification Task
def classification_example():
    # ساخت مدل ساده
    class SimpleClassifier(nn.Module):
        def __init__(self, input_size=784, num_classes=10):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)
            )
        
        def forward(self, x):
            return self.fc(x.view(x.size(0), -1))
    
    # ساخت dataset فیک
    train_data = torch.randn(1000, 784)
    train_labels = torch.randint(0, 10, (1000,))
    val_data = torch.randn(200, 784)
    val_labels = torch.randint(0, 10, (200,))
    
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Configuration
    config = TrainingConfig(
        model_name="simple_classifier",
        epochs=10,
        batch_size=32,
        optimizer=TrainingConfig.OptimizerConfig(
            name="adamw",
            lr=1e-3,
            weight_decay=0.01
        ),
        scheduler=TrainingConfig.SchedulerConfig(
            name="cosine",
            total_steps=1000
        ),
        logging=TrainingConfig.LoggingConfig(
            log_every=50,
            use_wandb=False,
            use_tensorboard=True
        ),
        checkpoint=TrainingConfig.CheckpointConfig(
            save_dir="./checkpoints/classification",
            save_every=5
        )
    )
    
    # Model
    model = SimpleClassifier()
    
    # Custom trainer برای classification
    from training_infra.trainer import ClassificationTrainer
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ]
    
    # Trainer
    trainer = ClassificationTrainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        callbacks=callbacks
    )
    
    # Train
    trainer.fit()
    
    # Save model
    trainer.save_model("./models/classifier.pt")

# مثال 2: Language Model Training
def language_model_example():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    
    # Load model and tokenizer
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Sample dataset
    texts = ["This is a sample text for training"] * 100
    
    # Tokenize
    def tokenize_batch(texts):
        return tokenizer(
            texts, 
            truncation=True, 
            padding=True, 
            max_length=128, 
            return_tensors="pt"
        )
    
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, texts, tokenizer, max_length=128):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = self.texts[idx]
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            input_ids = encoding["input_ids"].squeeze()
            labels = input_ids.clone()
            
            return input_ids, labels
    
    # Create datasets
    train_dataset = TextDataset(texts[:80], tokenizer)
    val_dataset = TextDataset(texts[80:], tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    # Configuration for language model
    config = TrainingConfig(
        model_name="gpt2_finetuned",
        epochs=5,
        batch_size=4,
        gradient_accumulation_steps=8,  # Effective batch size = 4 * 8 = 32
        max_grad_norm=1.0,
        optimizer=TrainingConfig.OptimizerConfig(
            name="adamw",
            lr=5e-5,
            weight_decay=0.01
        ),
        scheduler=TrainingConfig.SchedulerConfig(
            name="linear",
            warmup_steps=100,
            total_steps=500
        ),
        use_amp=True,
        amp_dtype="float16",
        logging=TrainingConfig.LoggingConfig(
            log_every=25,
            use_wandb=True,
            wandb_project="gpt2-finetuning"
        ),
        checkpoint=TrainingConfig.CheckpointConfig(
            save_dir="./checkpoints/gpt2",
            save_every=2,
            monitor="val_loss",
            mode="min"
        )
    )
    
    # Use language model trainer
    from training_infra.trainer import LanguageModelTrainer
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2)
    ]
    
    trainer = LanguageModelTrainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        callbacks=callbacks
    )
    
    trainer.fit()

# مثال 3: Distributed Training
def distributed_training_example():
    """مثال برای distributed training"""
    import os
    
    # Set environment variables for distributed training
    # این معمولاً توسط launcher (مثل torchrun) set می‌شه
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    # os.environ['RANK'] = '0'
    # os.environ['WORLD_SIZE'] = '2'
    
    config = TrainingConfig(
        model_name="distributed_model",
        epochs=10,
        batch_size=16,  # Per GPU batch size
        distributed=TrainingConfig.DistributedConfig(
            enabled=True,
            backend="nccl",
            find_unused_parameters=False
        ),
        logging=TrainingConfig.LoggingConfig(
            log_every=50,
            use_tensorboard=True
        )
    )
    
    # Model, data loaders, etc.
    model = SimpleClassifier()
    # ... setup data loaders
    
    # Training will automatically handle distributed setup
    trainer = Trainer(model, config, train_loader, val_loader)
    trainer.fit()

# مثال 4: Custom Loss Function
def custom_loss_example():
    """مثال با custom loss function"""
    
    class CustomTrainer(Trainer):
        def __init__(self, *args, loss_fn=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        
        def compute_loss(self, batch):
            inputs, targets = batch
            outputs = self.model(inputs)
            
            # Custom loss calculation
            base_loss = self.loss_fn(outputs, targets)
            
            # Add L2 regularization manually
            l2_reg = torch.tensor(0., device=self.device)
            for param in self.model.parameters():
                l2_reg += torch.norm(param)
            
            total_loss = base_loss + 0.01 * l2_reg
            
            # Log additional metrics
            if self.global_step % self.config.logging.log_every == 0:
                self.logger.log_metrics({
                    'train/base_loss': base_loss.item(),
                    'train/l2_reg': l2_reg.item(),
                    'train/total_loss': total_loss.item()
                }, self.global_step)
            
            return total_loss
        
        def validate(self):
            # Custom validation with additional metrics
            base_metrics = super().validate()
            
            if self.val_dataloader is None:
                return base_metrics
            
            # Add custom validation metrics
            self.model.eval()
            total_samples = 0
            correct_predictions = 0
            
            with torch.no_grad():
                for batch in self.val_dataloader:
                    batch = self._move_to_device(batch)
                    inputs, targets = batch
                    
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    total_samples += targets.size(0)
                    correct_predictions += (predicted == targets).sum().item()
            
            accuracy = correct_predictions / total_samples
            base_metrics['val/accuracy'] = accuracy
            
            return base_metrics
    
    # استفاده
    model = SimpleClassifier()
    config = TrainingConfig(epochs=10, batch_size=32)
    
    trainer = CustomTrainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        loss_fn=nn.CrossEntropyLoss(label_smoothing=0.1)
    )
    
    trainer.fit()

# مثال 5: Loading from Config File
def config_file_example():
    """مثال loading configuration از فایل"""
    
    # Save config to YAML
    config = TrainingConfig(
        model_name="example_model",
        epochs=20,
        batch_size=64,
        optimizer=TrainingConfig.OptimizerConfig(
            name="adamw",
            lr=1e-4,
            weight_decay=0.05
        )
    )
    
    config.save_yaml("config.yaml")
    
    # Load config from YAML
    loaded_config = TrainingConfig.from_yaml("config.yaml")
    
    print("Loaded config:", loaded_config.model_name)
    print("Learning rate:", loaded_config.optimizer.lr)

if __name__ == "__main__":
    print("Running classification example...")
    classification_example()
    
    print("\nRunning language model example...")
    # language_model_example()  # Uncomment if you have transformers installed
    
    print("\nRunning config file example...")
    config_file_example()