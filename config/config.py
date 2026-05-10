import ml_collections

def get_config():
  config = ml_collections.ConfigDict()

  # Model Save
  config.ResultPath = './Checkpoints/MADN/logs/'
  config.ModelSaveDir    = './Checkpoints/MADN/checkpoints/'


  # Data Info
  config.DataPath = '/mnt/data2/lintong/motion_correction/full_angle_tiny'
  config.Refine_DataPath = '/mnt/data2/lintong/motion_correction/full_angle_tiny'

  config.TrainBatchSize = 8
  config.FMKBatchSize = 8
  config.ImageShape = [512,512]

  # Train Info
  config.Epoch        = 500
  config.lr           = 1e-4
  config.beta1        = 0.9
  config.beta2        = 0.999
  config.weight_decay = 0.02
  config.StepSize     = 5
  config.Gamma        = 0.95
  config.Alpha        = 1
  config.use_cond     = True
  config.use_level    = False
  config.use_distance = False
  config.use_restored_cond = False
  # hyper-parameters
  config.eps = 1e-8

  return config
