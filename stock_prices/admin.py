from django.contrib import admin

# Register your models here.
from .models import post

class postModelAdmin(admin.ModelAdmin):
    list_display = [ "stock_name", "stock_symbol", "stock_price", "updated", "timestamp" ]
    list_display_links = [ "stock_name", "stock_symbol", "stock_price" ]
    list_display_links = [ "stock_name", "stock_symbol", "stock_price"]
    search_fields = [ "stock_name", "stock_symbol" ]
    class Meta:
        model = post

admin.site.register(post, postModelAdmin)
