import torch

from wenet.paraformer.embedding import ParaformerPositinoalEncoding


class SinusoidalPositionEncoder(torch.nn.Module):
    """https://github.com/alibaba-damo-academy/FunASR/blob/main/funasr/modules/embedding.py#L387
    """

    def __int__(self):
        pass

    def encode(self,
               positions: torch.Tensor,
               depth: int,
               dtype: torch.dtype = torch.float32) -> torch.Tensor:
        batch_size = positions.size(0)
        positions = positions.type(dtype)
        device = positions.device
        log_timescale_increment = torch.log(
            torch.tensor([10000], dtype=dtype,
                         device=device)) / (depth / 2 - 1)
        inv_timescales = torch.exp(
            torch.arange(depth / 2, device=device).type(dtype) *
            (-log_timescale_increment))
        inv_timescales = torch.reshape(inv_timescales, [batch_size, -1])
        scaled_time = torch.reshape(positions, [1, -1, 1]) * torch.reshape(
            inv_timescales, [1, 1, -1])
        encoding = torch.cat([torch.sin(scaled_time),
                              torch.cos(scaled_time)],
                             dim=2)
        return encoding.to(dtype)

    def forward(self, x):
        _, timesteps, input_dim = x.size()
        positions = torch.arange(1, timesteps + 1, device=x.device)[None, :]
        position_encoding = self.encode(positions, input_dim,
                                        x.dtype).to(x.device)
        return x + position_encoding, position_encoding


def test_pe():
    torch.manual_seed(777)
    Paraformer_pe = SinusoidalPositionEncoder()
    d_model = 512
    depth = 560
    Wenet_paraformer_pe = ParaformerPositinoalEncoding(d_model=d_model,
                                                       depth=560,
                                                       dropout_rate=0.0,
                                                       max_len=5000)
    Paraformer_pe.eval()
    Wenet_paraformer_pe.eval()

    input = torch.rand(1, 70, depth)
    paraformer_x_out, paraformer_pe_out = Paraformer_pe(input * d_model**0.5)
    wenet_paraformer_x_out, wenet_paraformer_pe_out = Wenet_paraformer_pe(
        input, offset=1)
    assert torch.allclose(paraformer_pe_out, wenet_paraformer_pe_out)
    assert torch.allclose(paraformer_x_out, wenet_paraformer_x_out)
