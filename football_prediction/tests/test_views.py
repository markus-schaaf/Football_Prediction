from django.test import TestCase
from django.urls import reverse

class TeamSelectionViewTest(TestCase):
    def test_team_selection_page_loads(self):
        response = self.client.get(reverse('team_selection'))
        self.assertEqual(response.status_code, 200)