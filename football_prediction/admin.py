from django.contrib import admin
from .models import Match

@admin.register(Match)
class MatchAdmin(admin.ModelAdmin):
    list_display = ('match_id', 'date', 'home_team', 'away_team', 'home_goals', 'away_goals', 'result', 'season')
    search_fields = ('home_team', 'away_team', 'season')
    list_filter = ('season', 'result')

from .models_views import (
    MatchWithAvgHomeGoals,
    MatchWithHomeWinRate,
    MatchWithAvgAwayGoals,
    MatchWithAwayWinRate,
)

@admin.register(MatchWithAvgHomeGoals)
class MatchWithAvgHomeGoalsAdmin(admin.ModelAdmin):
    list_display = ('match_id', 'home_team', 'home_goals', 'average_home_goals')

@admin.register(MatchWithHomeWinRate)
class MatchWithHomeWinRateAdmin(admin.ModelAdmin):
    list_display = ('match_id', 'home_team', 'home_goals', 'home_win_rate')

@admin.register(MatchWithAvgAwayGoals)
class MatchWithAvgAwayGoalsAdmin(admin.ModelAdmin):
    list_display = ('match_id', 'away_team', 'away_goals', 'average_away_goals')

@admin.register(MatchWithAwayWinRate)
class MatchWithAwayWinRateAdmin(admin.ModelAdmin):
    list_display = ('match_id', 'away_team', 'away_goals', 'away_win_rate')
