import torch
import timm

from einops import rearrange, pack

class ResNetDiscriminator(torch.nn.Module):
    def __init__(
            self,
            surface_variables = ["t2m", "u10", "v10", "msl"],
            upper_variables = ["u", "v", "t", "q", "z"],
            levels = [1000, 925, 850, 700, 500, 300, 150, 50],
            backbone_name = "resnet50",
            pretrained = False,
        ):
        super().__init__()

        self.surface_variables = surface_variables
        self.upper_variables = upper_variables
        self.levels = levels

        in_channels = len(self.surface_variables) + len(self.upper_variables) * len(self.levels)

        # Load a ResNet from timm, without the final classifier head
        self.backbone = timm.create_model(
            backbone_name,
            pretrained = pretrained,
            num_classes = 0,
            global_pool = 'avg',
        )
        
        # Replace first conv to accept custom channels
        self.backbone.conv1 = torch.nn.Conv2d(
            in_channels,
            self.backbone.conv1.out_channels,
            kernel_size = self.backbone.conv1.kernel_size,
            stride = self.backbone.conv1.stride,
            padding = self.backbone.conv1.padding,
            bias = False,
        )
        
        # Add discriminator head: outputs a probability (real/fake)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.backbone.num_features, 1),
            # torch.nn.Sigmoid(), # Only if using BCE loss
        )

    # def forward(self, x):
    #     features = self.backbone(x)
    #     out = self.fc(features)
    #     return out

    def map_var(self, var_name: str) -> str:
        _dict = {
            'u': 'u',
            'v': 'v',
            't': 't',
            'q': 'q',
            'z': 'z',
            't2m': '2t',
            'u10': '10u',
            'v10': '10v',
            'msl': 'msl',
            'lsm': 'lsm',
            'slt': 'slt',
        }
        if var_name in _dict:
            return _dict[var_name]
        else:
            raise ValueError(f"Variable {var_name} not found in mapping dictionary.")

    def _forward_features(self, batch) -> torch.Tensor:
        # pass

        # Access variable value and then combine them into a tensor.

        p = next(self.parameters())
        batch = batch.type(p.dtype)
        batch = batch.normalise(surf_stats = dict())
        # batch = batch.crop(patch_size=self.patch_size)
        batch = batch.to(p.device)

        atmos, _ = pack( [batch.atmos_vars[ self.map_var(v) ] for v in self.upper_variables], "b * l h w" )
        surf, _ = pack( [batch.surf_vars[ self.map_var(v) ] for v in self.surface_variables], "b * h w" )
    
        all_var_tensor, _ = pack( [rearrange( atmos, "b v l h w -> b (v l) h w" ), surf], "b * h w")
        
        x = all_var_tensor

        features = self.backbone(x)

        return features

    def forward(self, batch) -> torch.Tensor:
        # pass

        # Access variable value and then combine them into a tensor.

        # p = next(self.parameters())
        # batch = batch.type(p.dtype)
        # batch = batch.normalise(surf_stats = dict())
        # batch = batch.to(p.device)

        # # print(f"{batch.static_vars['z'].device=}")

        # atmos, _ = pack( [batch.atmos_vars[ self.map_var(v) ] for v in self.upper_variables], "b * l h w" )
        # surf, _ = pack( [batch.surf_vars[ self.map_var(v) ] for v in self.surface_variables], "b * h w" )
    
        # all_var_tensor, _ = pack( [rearrange( atmos, "b v l h w -> b (v l) h w" ), surf], "b * h w")
        
        # x = all_var_tensor

        # features = self.backbone(x)
        features = self._forward_features(batch)
        output = self.fc(features)

        return output