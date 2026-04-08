Taramux certbot
===============

docker compose run --rm -p 80:80 -p 443:443 certbot certbot certonly --standalone --non-interactive --agree-tos --deploy-hook /deploy-hook.sh -d [DOMAIN]
