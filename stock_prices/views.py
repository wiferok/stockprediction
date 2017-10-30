from django.shortcuts import render

# create views here
from .models import post

def index(request):
    context = {
        "":"",
    }
    return render(request, 'index.html', context)
