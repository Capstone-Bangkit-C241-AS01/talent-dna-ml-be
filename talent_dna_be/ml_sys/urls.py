from django.urls import path
from .views import simple_api_view, ml_process

urlpatterns = [
    path('test/', simple_api_view, name='simple_api_view'),
    path('', ml_process, name='ml_process')
]
