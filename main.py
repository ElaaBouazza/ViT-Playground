import hydra
from omegaconf import DictConfig
from src.data import get_dataloaders
from src.modeling import ViTransformer
import numpy as np
import pytorch_lightning as pl

@hydra.main(version_base=None, config_path="conf", config_name="config")

def main(cfg: DictConfig):
    pl.seed_everything(cfg.training.seed)
    
    model = ViTransformer(cfg)
    train_loader, val_loader, test_loader = get_dataloaders(cfg.data.img_size, cfg.data.batch_size)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='valid_acc',
        mode='max',
        save_top_k=1,
        filename='best-checkpoint'
    )
    """
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        precision=cfg.training.precision,
        max_epochs=cfg.training.max_epochs,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
    ) 
    """
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs, 
        accelerator= 'gpu',
        devices = 1, 
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_loader, val_loader)
    test_result = trainer.test(model, test_loader, ckpt_path='best')
    print(f"Test accuracy: {test_result[0]['test_acc']:.4f}")
    # update_readme(test_result[0]['test_acc'])

if __name__ == "__main__":
    main()