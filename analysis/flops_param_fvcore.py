import torch
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from model_zoo.lsm import buildLSMS, buildLSM
from model_zoo.lsm_light import buildLSM_light
from model_zoo.swinIR import buildSwinIR, buildSwinIR_light
from model_zoo.mambaIR import buildMambaIR, buildMambaIR_light
from model_zoo.mambairv2 import buildMambaIRv2S, buildMambaIRv2B
from model_zoo.mambairv2light import buildMambaIRv2Light
from model_zoo.atd import buildATD, buildATD_light
from model_zoo.cat import buildCATR, buildCATA
from model_zoo.dat import buildDATS, buildDAT


from analysis.utils_fvcore import FLOPs

fvcore_flop_count = FLOPs.fvcore_flop_count

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H=512
    W=512
    scale=4
    init_model = buildATD_UNet(upscale=scale).to(device)
    with torch.no_grad():
        FLOPs.fvcore_flop_count(init_model, input_shape=(3, H//scale,W//scale))
        