from django.urls import path, include
from . import views
from TrOCR_Django_App.views import ocr_task_view
app_name = 'Deep_Learning_App'

urlpatterns = [
    path('show_preprocessed_batch/', views.show_preprocessed_batch, name='show_preprocessed_batch'),
    path('configure/', views.configure, name='configure'),
    path('train_model/', views.train_model, name='train_model'),
    path('see_logs/', views.see_logs, name='see_logs'),
    
]