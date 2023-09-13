from django.shortcuts import render

# Create your views here.
def ocr_task_view(request):
    # Your view logic goes here
    return render(request, 'ocr_task.html')