from django.shortcuts import render
from django.http import HttpResponse
import subprocess, os, sys
from Deep_Learning_App.forms import ConfigForm
from Deep_Learning_App.src.config import config
from Deep_Learning_App.data_handler.data_splitter import DataSplitter
from Deep_Learning_App.data_handler.data_loader import CustomDataset
import tensorflow as tf
import random
from PIL import Image
from transformers import GPT2Tokenizer  # Import the GPT2Tokenizer from the Hugging Face Transformers library


# Create your views here.
def show_preprocessed_batch(request):
        tokenizer = GPT2Tokenizer.from_pretrained("akhooli/gpt2-small-arabic")

        data_splitter = DataSplitter(config.data_path, 0.3, config.train_test_validation_ratios[2]) # For testing purpose, we only sample a small set

        train_paths = data_splitter.get_train_paths()
        randomly_sampled_paths = random.sample(train_paths, config.batch_size)
        # training_images_paths = [path+".jpg" for path in randomly_sampled_paths]
        # training_labels_paths = [path+".txt" for path in randomly_sampled_paths] # .replace('\\\\', '\\')
        
        # Since web browsers doesn't allow browsers to access local files, absolute paths won't show up
        # Relative paths works
        
        # context = []
        # context.append({'path': training_images_paths[i], 'label': training_labels_paths[i]})


        # for i in range(len(randomly_sampled_paths)):
        #     index = randomly_sampled_paths[i].find('media')  # Find the index of the subword in the input string
        #     randomly_sampled_paths[i] = randomly_sampled_paths[i][index -1 :]  # Delete everything before the subword
        # print('randomly_sampled_paths', randomly_sampled_paths)
        
        dataset_generator = CustomDataset(randomly_sampled_paths)
        dataset_generator.batch_size = len(randomly_sampled_paths)

        context = []

        # Get a batch of preprocessed image
        for inputs, outputs in dataset_generator:
            for i in range(inputs[0].shape[0]):
                tensor_image = tf.cast(inputs[0][i]*255, tf.uint8)
                output_text = tokenizer.decode(outputs[i], skip_special_tokens=True)  # Shape: `()`.
                # print("\n\nTracing:", output_text)

                # Convert the TensorFlow tensor to a NumPy array
                numpy_array = tensor_image.numpy()

                # Create a PIL image from the NumPy array
                pil_image = Image.fromarray(numpy_array)
                preprocessed_path = os.path.join(config.data_path, str(i)+'_edited.jpg')
                # print("Saving path:", path)
                pil_image.save(preprocessed_path)

                
                preprocessed_image_index = preprocessed_path.find('media')
                preprocessed_image__index = preprocessed_path.find('dataset')
                preprocessed_path = preprocessed_path[preprocessed_image_index -1 :preprocessed_image__index+len('dataset')+1]

                original_image_index = randomly_sampled_paths[i].find('media')  # Find the index of the subword in the input string
                randomly_sampled_paths[i] = randomly_sampled_paths[i][original_image_index -1 :]  # Delete everything before the subword

                context.append({'original_path': randomly_sampled_paths[i]+".jpg",
                            'processed_path': preprocessed_path+str(i)+"_edited.jpg",
                            'label': output_text})
            break
        
        # print('context', context)
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