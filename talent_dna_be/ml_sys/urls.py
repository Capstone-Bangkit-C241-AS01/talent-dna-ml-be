from django.urls import path
from .views import simple_api_view, ml_process, get_all_responses

urlpatterns = [
    path('test/', simple_api_view, name='simple_api_view'),
    path('', ml_process, name='ml_process'),
    path('data/', get_all_responses, name='get_all_responses')
]
