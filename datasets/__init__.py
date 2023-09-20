from datasets.csl_500.dataloader_csl_isolated import CSL_ISO
from datasets.csl_split1.dataloader_csl_split1 import CSL_SPLIT1
from datasets.csl_split2.dataloader_csl_split2 import CSL_SPLIT2
from datasets.gsl_si.dataloader_gsl_si import GSL_SI
from datasets.gsl_isolated.dataloader_greek_isolated import GSL_ISO
from datasets.gsl_sd.dataloader_gsl_sd import GSL_SD
from datasets.phoenix2014.dataloader_ph2014features import PHOENIX2014_FEATS
from datasets.phoenix2014.dataloader_phoenix2014 import PHOENIX2014
from datasets.phoenix2014T.dataloader_phoenix2014_T import PHOENIX2014T
from datasets.phoenix2014.dataloader_phoenix2014_cui_augmentations import PH2014_CUI_AUG
from datasets.phoenix2014isolated.dataloader_phoenix2014_isolated import PHOENIX2014_ISO
from datasets.phoenix2014isolated.dataloader_phoenix2014_isolated_cui import PH2014_ISO_CUI
from datasets.phoenix2014I5.dataloader_phoenix2014_signer5_continuous import PHOENIX_I5
from .dataloader_phoenix2014_signer5_isolated import PHOENIX_I5_ISO
from datasets.ms_asl.dataloader_ms_asl import MSASL_Dataset
from .loader_utils import select_continouous_dataset,select_isolated_dataset,select_scenario_for_training


