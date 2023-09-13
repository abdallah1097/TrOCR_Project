from django import forms
from Deep_Learning_App.src.config import config

class ConfigForm(forms.Form):
    data_path = forms.CharField(max_length=100, initial=config.data_path, help_text='Dataset directory path')
    deep_learning_model_path = forms.CharField(max_length=100, initial=config.deep_learning_model_path, help_text='DeepLearning Model directory path')
    train_ratio = forms.FloatField(initial=config.train_test_validation_ratios[0], help_text='e.g. 0.15 not 15%')
    validation_ratio = forms.FloatField(initial=config.train_test_validation_ratios[1], help_text='e.g. 0.15 not 15%')
    test_ratio = forms.FloatField(initial=config.train_test_validation_ratios[2], help_text='e.g. 0.15 not 15%')
    split_random_seed = forms.IntegerField(initial=config.split_random_seed)
    batch_size = forms.IntegerField(initial=config.batch_size, help_text='Better be multiples of 2^(.). Typically: 8,16,24,32,...')
    image_height = forms.IntegerField(initial=config.image_height)
    image_width = forms.IntegerField(initial=config.image_width)
    patch_height = forms.IntegerField(initial=config.patch_height)
    patch_width = forms.IntegerField(initial=config.patch_width)
    d_model = forms.IntegerField(initial=config.d_model, help_text='d_model')
    dff = forms.IntegerField(initial=config.dff, help_text='dff')
    num_heads = forms.IntegerField(initial=config.num_heads, help_text='Encoder/ Decoder number of Attention Heads')
    num_layers = forms.IntegerField(initial=config.num_layers, help_text='Encoder/ Decoder number of Number of Attention Layers')
    max_length = forms.IntegerField(initial=config.max_length, help_text='Maximum length to pad')