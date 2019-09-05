






FROM continuumio/anaconda3:4.4.0
MAINTAINER UNP, https://unp.education
EXPOSE 8000
RUN echo "deb http://deb.debian.org/debian jessie main" > /etc/apt/sources.list
RUN apt-get update && apt-get install -y apache2 \
    apache2-dev \
    vim \
 && apt-get clean \
 && apt-get autoremove \
 && rm -rf /var/lib/apt/list/*
WORKDIR /var/www/dogcat_api/
COPY requirements.txt .
COPY ./img_reco_test.wsgi /var/www/dogcat_api/img_reco_test.wsgi
COPY ./dogcat_flaskDemo  /var/www/dogcat_api/
RUN pip install -r requirements.txt 
RUN /opt/conda/bin/mod_wsgi-express install-module
RUN mod_wsgi-express setup-server img_reco_test.wsgi --port=8000 \
    --user www-data --group www-data \
    --server-root=/etc/mod_wsgi-express-80
CMD /etc/mod_wsgi-express-80/apachectl start -D FOREGROUND