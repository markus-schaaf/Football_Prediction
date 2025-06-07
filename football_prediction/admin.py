from django.contrib import admin
from .models import Match

@admin.register(Match)
class MatchAdmin(admin.ModelAdmin):
    list_display = ('match_id', 'date', 'home_team', 'away_team', 'home_goals', 'away_goals', 'result', 'season')
    search_fields = ('home_team', 'away_team', 'season')
    list_filter = ('season', 'result')
