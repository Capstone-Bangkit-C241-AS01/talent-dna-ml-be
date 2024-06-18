from rest_framework import serializers
from .models import Top10Talent, Bottom5Talent, JobRecommendation, Users


class Top10TalentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Top10Talent
        fields = '__all__'


class Bottom5TalentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Bottom5Talent
        fields = '__all__'


class JobRecommendationSerializer(serializers.ModelSerializer):
    class Meta:
        model = JobRecommendation
        fields = '__all__'


class UsersSerializer(serializers.ModelSerializer):
    top_10_talents = Top10TalentSerializer(many=True, read_only=True)
    bottom_5_talents = Bottom5TalentSerializer(many=True, read_only=True)
    job_recommendations = JobRecommendationSerializer(
        many=True, read_only=True)

    class Meta:
        model = Users
        fields = ['id', 'name', 'top_talent_description', 'bottom_talent_description',
                  'top_10_talents', 'bottom_5_talents', 'job_recommendations']
