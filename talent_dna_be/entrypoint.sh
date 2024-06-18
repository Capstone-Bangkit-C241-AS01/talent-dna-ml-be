#!/bin/sh

if [ "$DATABASE" = "postgres" ]
then
    echo "Tunggu database postgres siap..."

    while ! nc -z $DB_HOST $DB_PORT; do
      sleep 0.1
    done

    echo "Database siap"
fi
echo "Checking environment variables"
echo "DATABASE: $DATABASE"
echo "DB_HOST: $DB_HOST"
echo "DB_PORT: $DB_PORT"
# perform database migration
python manage.py makemigrations
python manage.py migrate

# start the Django server
exec "$@"