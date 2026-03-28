from src.config.config_loader import load_and_validate_config
from src.data.compose.composer import DatasetComposer
from src.models.compose.composer import ModelComposer
from src.training.trainer import Trainer
from src.quantization.quantizer import Quantizer

config = load_and_validate_config("config/rniq_config_resnet34_cifar10.yaml")

dataset_composer = DatasetComposer(config=config)
model_composer = ModelComposer(config=config)
quantizer = Quantizer(config=config)()
trainer = Trainer(config=config)

data = dataset_composer.compose()
model = model_composer.compose()

# Validate  model before quantization
trainer.validate(model, datamodule=data)

qmodel = quantizer.quantize(model, in_place=True)

# Validate model after layers replacement
trainer.validate(qmodel, datamodule=data)

# Calibrating model initial weights and scales if defined in config
trainer.calibrate(qmodel, datamodule=data)

# Finetune model
trainer.fit(qmodel, datamodule=data)

# Test model after quantization
trainer.test(qmodel, datamodule=data, ckpt_path="best")
