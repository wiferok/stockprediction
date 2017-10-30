from django.db import models

# Create your models here.
#MVC

class post(models.Model):
    stock_name = models.CharField(max_length=120)
    stock_symbol = models.CharField(max_length=120)
    stock_price = models.CharField(max_length=120)
    updated = models.DateTimeField(auto_now=True, auto_now_add=False)
    timestamp = models.DateTimeField(auto_now=False, auto_now_add=True)

def _str_(self):
    return self.title

def _unicode_(self):
    return self.title
