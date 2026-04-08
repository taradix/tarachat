ARG NGINX_VERSION=1.29.4

FROM nginx:${NGINX_VERSION}-alpine

RUN rm /etc/nginx/conf.d/default.conf

COPY conf.d/ /etc/nginx/conf.d/
COPY nginx.conf /etc/nginx/nginx.conf
