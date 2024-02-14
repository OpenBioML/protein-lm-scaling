import optuna
from mup import make_base_shapes, set_base_shapes, make_base_shapes, set_base_shapes, get_shapes, MuAdam, MuSGD, MuAdamW
from mup.coord_check import get_coord_data, plot_coord_data
from functools import partial
import torch
from protein_lm.modeling import APTConfig, APTLMHeadModel




# not sure how to leverage pytest in the context of coordinate checking
# this is because visual inspection of coordinate checking results is necessary
# the test will generate coordinate checking results to a test_results directory for now

if __name__ == "__main__":
    delta_model = APTLMHeadModel(config=APTConfig(n_embd=200, n_layer=8, num_attention_heads=10, n_inner=200, use_mup=True))
    delta_model.apply(delta_model._init_weights)

    base_model = APTLMHeadModel(config=APTConfig(n_embd=1, n_layer=8, num_attention_heads=1, n_inner=1, use_mup=True))
    base_model.apply(base_model._init_weights)

    def get_mup_apt_model(width):
        model = APTLMHeadModel(config=APTConfig(n_embd=width, n_layer=8, num_attention_heads=width//16, n_inner=width, use_mup=True))
        return model
    
    def set_up_mup_apt_model(width):
        model = set_base_shapes(get_mup_apt_model(width), base_model, delta=delta_model)
        model.apply(model._init_weights)
        return model
    
    def get_mup_lazy_model(width):
        return lambda: set_up_mup_apt_model(width)
    
    models = {256: get_mup_lazy_model(256), 512: get_mup_lazy_model(512), 1024: get_mup_lazy_model(1024), 2048: get_mup_lazy_model(2048)}

    input_ids = torch.randint(low=0, high=50257, size=(1, 256)).to(torch.int64)
    labels = torch.randint(low=0, high=50257, size=(1, 256)).to(torch.int64)
    dataloader=[{'input_ids': input_ids, 'labels': labels}]
    df = get_coord_data(models, dataloader, optimizer='sgd', lr=0.1, dict_in_out=True, output_name='loss', cuda=True, nsteps=10, nseeds=10)
    plot_coord_data(df, legend=None, save_to='test_results/apt_coordcheck.jpg')