from rest_framework import serializers
from .models import Response, Top10Talent, Bottom5Talent, JobRecommendation


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


class ResponseSerializer(serializers.ModelSerializer):
    top_10_talents = Top10TalentSerializer(many=True)
    bottom_5_talents = Bottom5TalentSerializer(many=True)
    job_recommendations = JobRecommendationSerializer(many=True)

    class Meta:
        model = Response
        fields = '__all__'
