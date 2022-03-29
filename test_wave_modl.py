##########################################################
# %%
# tensorflow later than 2.2.0
##########################################################
import os,sys
os.environ["CUDA_DEVICE_ORDER"]     = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]  = "0"
sys.path.append('utils')

##########################################################
# %%
# import libraries
##########################################################
import  numpy               as  np
import  library_common      as  mf
import  library_modl        as  mm
import  library_wave        as  mw

##########################################################
# %%
# data download
##########################################################
if os.path.exists('data/data.npz')==False:
    os.system("wget -O data/data.npz https://www.dropbox.com/s/vs8v3blc1v08fv0/data.npz?dl=1")

##########################################################
# %%
# parameters
##########################################################
ry          = 4
rz          = 3

Tadc        = 2880*1e-6         # sec
FOVy        = 240*1e-3          # m
FOVz        = 192*1e-3          # m
Gwy         = 16.5118*1e-3      # T/m
Cwy         = 5                 # of cycle
Gwz         = 16.5118*1e-3      # T/m
Cwz         = 5                 # of cycle
zpadf       = 1.2               # fixed value 

num_block   = 10
nLayers     = 5
num_filters = 24

num_slc, nx, ny, nc, ne     = 192, 480, 238, 32, 5
w_contrast                  = [3.26, 2.36, 1.57 , 1.12, 1]

##########################################################
# %%
# load data
##########################################################
data    =   np.load('data/data.npz')
ref         =   data['ref']
csm         =   data['csm']
psf_wav     =   data['psf_wav']
mask_caipi  =   data['mask_caipi']
mask_fixed  =   data['mask_fixed']
input_sens      =   data['Atb0']
input_modl      =   data['Atb1']
input_wav_caipi =   data['Atb2']
input_wav_modl  =   data['Atb3']

##########################################################
# %%
# create networks
##########################################################
# sense
model_sens = mm.create_sense(nx, ny, rz, nc, ne, nLayers, num_block)
model_sens.compile(optimizer=[],loss=mf.nrmse_loss) 
# modl
model_modl = mm.create_modl(nx, ny, rz, nc, ne, nLayers, num_block, num_filters)
model_modl.compile(optimizer=[],loss=mf.nrmse_loss) 
# wave caipi
model_wav_caipi = mw.create_caipi(nx, ny, rz, nc, ne, nLayers, num_block, num_filters, zpadf)
model_wav_caipi.compile(optimizer=[],loss=mf.nrmse_loss) 
# wave modl
model_wav_modl  = mw.create_wave(nx, ny, rz, nc, ne, nLayers, num_block, num_filters, zpadf)
model_wav_modl.compile(optimizer=[],loss=mf.nrmse_loss) 

##########################################################
# %%
# loading the network
##########################################################
model_modl.load_weights('network/modl.hdf5')
model_wav_modl.load_weights('network/wave_modl.hdf5')

##########################################################
# %%
# Prediction
##########################################################
# model prediction
P_sens          =  model_sens.predict([csm,mask_fixed[:,:nx,],input_sens])
P_modl          =  model_modl.predict([csm,mask_caipi[:,:nx,],input_modl])
P_wave_caipi    =  model_wav_caipi.predict([csm,mask_fixed,psf_wav,input_wav_caipi])
P_wave_modl     =  model_wav_modl.predict([csm,mask_caipi,psf_wav,input_wav_modl])

# reshape
I_sens          =   np.abs(mf.r2c5(P_sens).numpy()[0,int(0.25*nx):int(0.75*nx),])
I_modl          =   np.abs(mf.r2c5(P_modl).numpy()[0,int(0.25*nx):int(0.75*nx),])
I_wave_caipi    =   np.abs(mf.r2c5(P_wave_caipi).numpy()[0,int(0.25*nx):int(0.75*nx),])
I_wave_modl     =   np.abs(mf.r2c5(P_wave_modl).numpy()[0,int(0.25*nx):int(0.75*nx),])

# nrmse
L_sens          =   mf.nrmse_loss(ref, P_sens).numpy()
L_modl          =   mf.nrmse_loss(ref, P_modl).numpy()
L_wave_caipi    =   mf.nrmse_loss(ref, P_wave_caipi).numpy()
L_wave_modl     =   mf.nrmse_loss(ref, P_wave_modl).numpy()

##########################################################
# %%
# diplay results
##########################################################
mf.mosaic(I_sens[:,:,1,:],1,ne,1,[0,1],'SENSE NRMSE : %.2f' % L_sens)
mf.mosaic(I_modl[:,:,1,:],1,ne,2,[0,1],'MoDL NRMSE : %.2f' % L_modl)
mf.mosaic(I_wave_caipi[:,:,1,:],1,ne,3,[0,1],'Wave-CAIPI NRMSE : %.2f' % L_wave_caipi)
mf.mosaic(I_wave_modl[:,:,1,:],1,ne,4,[0,1],'Wave MoDL NRMSE : %.2f' % L_wave_modl)

##########################################################
# %%
# save results
##########################################################
np.savez('results/result.npz', I_sens = I_sens, I_modl = I_modl, I_wave_caipi = I_wave_caipi, I_wave_modl = I_wave_modl )