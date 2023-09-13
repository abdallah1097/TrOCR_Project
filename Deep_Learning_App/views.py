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
def show_statistics(request):
        data_splitter = DataSplitter(config.data_path, config.train_test_validation_ratios[1], config.train_test_validation_ratios[2])

        train_paths = data_splitter.get_train_paths()
        val_paths = data_splitter.get_val_paths()
        test_paths = data_splitter.get_test_paths()

        training_images = [path+".jpg" for path in random.sample(train_paths, 5)]
        testing_images = [path+".jpg" for path in random.sample(val_paths, 5)]
        validation_images = [path+".jpg" for path in random.sample(test_paths, 5)]

        # Create a context dictionary to pass data to the template
        context = {
            'training_count': len(train_paths),
            'testing_count': len(val_paths),
            'validation_count': len(test_paths),
            'training_images': training_images,
            'testing_images': testing_images,
            'validation_images': validation_images,
        }

        return render(request, 'show_statistics.html', context)


# Create your views here.
def start_tensorboard(request):
    if request.method == 'POST':
        port = request.POST.get('port')
        logdir = request.POST.get('logdir')
        try:
            # Start TensorBoard as a subprocess
            subprocess.Popen(['tensorboard', '--logdir', logdir, '--port', port])
            return HttpResponse(f"TensorBoard started on port {port} with log directory: {logdir}")
        except Exception as e:
            # return HttpResponse(f"TensorBoard started on port {port} with log directory: {logdir}")
            # del port, logdir
            return HttpResponse(f"Error starting TensorBoard: {str(e)}")
    else:
        return render(request, 'start_tensorboard.html')

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