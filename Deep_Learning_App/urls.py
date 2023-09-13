from django.urls import path, include
from . import views
from TrOCR_Django_App.views import ocr_task_view
app_name = 'Deep_Learning_App'

urlpatterns = [
    path('start_tensorboard/', views.start_tensorboard, name='start_tensorboard'),
    path('configure/', views.configure, name='configure'),
    path('train_model/', views.train_model, name='train_model'),
    path('show_statistics/', views.show_statistics, name='show_statistics'),
    
]