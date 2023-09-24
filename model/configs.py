import json
from pydantic import BaseModel, validator, root_validator
from typing import List, Iterable, Optional, Union, Tuple, Dict, Any
from enum import Enum

#from imagen_pytorch import Unet, Unet3D, NullUnet
from trainer import ImagenTrainer
from infer import Elucidated_Diffusion_Model
from bert import DEFAULT_BERT_NAME, get_encoded_dim

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def ListOrTuple(inner_type):
    return Union[List[inner_type], Tuple[inner_type]]

def SingleOrList(inner_type):
    return Union[inner_type, ListOrTuple(inner_type)]

# noise schedule

class NoiseSchedule(Enum):
    cosine = 'cosine'
    linear = 'linear'

class AllowExtraBaseModel(BaseModel):
    class Config:
        extra = "allow"
        use_enum_values = True

# imagen pydantic classes

class NullUnetConfig(BaseModel):
    is_null:            bool

    def create(self):
        return NullUnet()

class UnetConfig(AllowExtraBaseModel):
    dim:                int
    dim_mults:          ListOrTuple(int)
    text_embed_dim:     int = get_encoded_dim(DEFAULT_BERT_NAME)
    cond_dim:           int = None
    channels:           int = 3
    attn_dim_head:      int = 32
    attn_heads:         int = 16

    def create(self):
        return Unet(**self.dict())

class Unet3DConfig(AllowExtraBaseModel):
    dim:                int
    dim_mults:          ListOrTuple(int)
    text_embed_dim:     int = get_encoded_dim(DEFAULT_BERT_NAME)
    cond_dim:           int = None
    channels:           int = 3
    attn_dim_head:      int = 32
    attn_heads:         int = 16

    def create(self):
        return Unet3D(**self.dict())

class ElucidatedImagenConfig(AllowExtraBaseModel):
    unets:                  ListOrTuple(Union[UnetConfig, Unet3DConfig, NullUnetConfig])
    image_sizes:            ListOrTuple(int)
    video:                  bool = False
    text_encoder_name:      str = DEFAULT_BERT_NAME
    channels:               int = 3
    cond_drop_prob:         float = 0.5
    num_sample_steps:       SingleOrList(int) = 32
    sigma_min:              SingleOrList(float) = 0.002
    sigma_max:              SingleOrList(int) = 80
    sigma_data:             SingleOrList(float) = 0.5
    rho:                    SingleOrList(int) = 7
    P_mean:                 SingleOrList(float) = -1.2
    P_std:                  SingleOrList(float) = 1.2
    S_churn:                SingleOrList(int) = 80
    S_tmin:                 SingleOrList(float) = 0.05
    S_tmax:                 SingleOrList(int) = 50
    S_noise:                SingleOrList(float) = 1.003

    @validator('image_sizes')
    def check_image_sizes(cls, image_sizes, values):
        unets = values.get('unets')
        if len(image_sizes) != len(unets):
            raise ValueError(f'image sizes length {len(image_sizes)} must be equivalent to the number of unets {len(unets)}')
        return image_sizes

    def create(self):
        decoder_kwargs = self.dict()
        unets_kwargs = decoder_kwargs.pop('unets')
        is_video = decoder_kwargs.pop('video', False)

        unet_klass = Unet3D if is_video else Unet

        unets = []

        for unet, unet_kwargs in zip(self.unets, unets_kwargs):
            if isinstance(unet, NullUnetConfig):
                unet_klass = NullUnet
            elif is_video:
                unet_klass = Unet3D
            else:
                unet_klass = Unet

            unets.append(unet_klass(**unet_kwargs))

        imagen = Elucidated_Diffusion_Model(unets, **decoder_kwargs)

        imagen._config = self.dict().copy()
        return imagen
