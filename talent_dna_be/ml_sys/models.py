import uuid
from django.db import models


class Top10Talent(models.Model):
    name = models.CharField(max_length=100)
    predicted_rank = models.FloatField()
    strength = models.FloatField()


class Bottom5Talent(models.Model):
    name = models.CharField(max_length=100)
    predicted_rank = models.FloatField()
    strength = models.FloatField()


class JobRecommendation(models.Model):
    job = models.CharField(max_length=100)
    distance = models.FloatField()
    tasks = models.TextField()
    work_styles = models.TextField()


class Users(models.Model):
    name = models.CharField(max_length=100)
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    top_talent_description = models.TextField()
    bottom_talent_description = models.TextField()
    top_10_talents = models.ManyToManyField(Top10Talent)
    bottom_5_talents = models.ManyToManyField(Bottom5Talent)
    job_recommendations = models.ManyToManyField(JobRecommendation)
