from django.shortcuts import render
from django.http import HttpResponse
import subprocess, os, sys
from Deep_Learning_App.forms import ConfigForm
from Deep_Learning_App.src.config import config
from Deep_Learning_App.data_handler.data_splitter import DataSplitter
from Deep_Learning_App.data_handler.data_splitter import DataSplitter
from Deep_Learning_App.data_handler.data_splitter import DataSplitter
import random

# Create your views here.
def show_preprocessed_batch(request):
        data_splitter = DataSplitter(config.data_path, 0.1, config.train_test_validation_ratios[2]) # For testing purpose, we only sample a small set

        train_paths = data_splitter.get_train_paths()
        randomly_sampled_paths = random.sample(train_paths, 5)
        training_images_paths = [path+".jpg" for path in randomly_sampled_paths]
        training_labels_paths = [path+".txt" for path in randomly_sampled_paths] # .replace('\\\\', '\\')
        
        # Since web browsers doesn't allow browsers to access local files, absolute paths won't show up
        # Relative paths works
        
        context = []

        for i in range(len(training_images_paths)):
            index = training_images_paths[i].find('media')  # Find the index of the subword in the input string
            training_images_paths[i] = training_images_paths[i][index -1 :]  # Delete everything before the subword

            index = training_labels_paths[i].find('media')  # Find the index of the subword in the input string
            training_labels_paths[i] = training_labels_paths[i][index -1 :]  # Delete everything before the subword
            context.append({'path': training_images_paths[i], 'label': training_labels_paths[i]})

        return render(request, 'show_image_preprocessing.html', {'context': context})


# Create your views here.
def see_logs(request):
    # Start TensorBoard as a subprocess
    process = subprocess.Popen(['tensorboard', '--logdir', config.log_dir])
    # process.wait()

    return render(request, 'see_logs.html', {'config': {'log_dir': config.log_dir}})

        

def configure(request):
    if request.method == 'POST':
        form = ConfigForm(request.POST)
        print("This is a log message")
        if form.is_valid():
            print("This is a log message")
            # Update the config.py variables with the new values
            config.data_path = form.cleaned_data['data_path']
            config.deep_learning_model_path = form.cleaned_data['deep_learning_model_path']
            config.train_test_validation_ratios = [form.cleaned_data['train_ratio'], form.cleaned_data['validation_ratio'], form.cleaned_data['test_ratio']]
            config.split_random_seed = form.cleaned_data['split_random_seed']
            config.batch_size = form.cleaned_data['batch_size']
            config.image_height = form.cleaned_data['image_height']
            config.image_width = form.cleaned_data['image_width']
            config.patch_height = form.cleaned_data['patch_height']
            config.patch_width = form.cleaned_data['patch_width']
            config.d_model = form.cleaned_data['d_model']
            config.dff = form.cleaned_data['dff']
            config.num_heads = form.cleaned_data['num_heads']
            config.num_layers = form.cleaned_data['num_layers']
            config.max_length = form.cleaned_data['max_length']
            # Save the changes if necessary

            # Redirect to a success page or render a response
            # (e.g., show updated configuration or a confirmation message)
            return render(request, 'success_form.html')
        else:
            # Form is invalid, display error messages
            print(form.errors)
    else:
        form = ConfigForm()

    return render(request, 'config_form.html', {'form': form})


def train_model(request):
    venv_python = sys.executable
    subprocess_cmd = [venv_python, 'deep_learning/src/train.py']
    working_directory = os.getcwd()

    # change_dir_cmd = ['cd', working_directory]
    # subprocess.Popen(change_dir_cmd, shell=True).wait()  # Use shell=True for the cd command

    process = subprocess.Popen(subprocess_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, cwd=working_directory, shell=True)

    # Start TensorBoard as a subprocess
    command = f'tensorboard --logdir={config.log_dir}'
    process = subprocess.Popen(command, shell=True)


    # stdout, stderr = process.communicate()
    # return render(request, 'training.html', {'stdout': stdout, 'stderr': stderr})
    return render(request, 'training.html')