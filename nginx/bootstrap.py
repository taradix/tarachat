import os
import subprocess
from pathlib import Path
from textwrap import dedent

from jinja2 import Environment, FileSystemLoader


def includes_conf(env, template_vars):
    conf_d = Path("/etc/nginx/conf.d")
    conf_d.joinpath("server_name.active").write_text(dedent(f"""
        server_name {template_vars['SERVER_HOSTNAME']} autodiscover.* autoconfig.* {' '.join(template_vars['ADDITIONAL_SERVER_NAMES'])};
    """))

def sites_default_conf(env, template_vars):
    path = Path("/etc/nginx/includes/sites-default.conf")
    template = env.get_template(f"{path.name}.j2")
    config = template.render(template_vars)
    path.write_text(config)

def nginx_conf(env, template_vars):
    path = Path("/etc/nginx/nginx.conf")
    template = env.get_template(f"{path.name}.j2")
    config = template.render(template_vars)
    path.write_text(config)

def prepare_template_vars(environ):
    ipv4_network = environ.get("IPV4_NETWORK", "172.22.1")
    additional_server_names = environ.get("ADDITIONAL_SERVER_NAMES", "")

    template_vars = {
        "IPV4_NETWORK": ipv4_network,
        "TRUSTED_NETWORK": environ.get("TRUSTED_NETWORK", False),
        "SERVER_HOSTNAME": environ.get("SERVER_HOSTNAME", ""),
        "ADDITIONAL_SERVER_NAMES": [item.strip() for item in additional_server_names.split(",") if item.strip()],
    }

    ssl_dir = Path("/etc/letsencrypt/live/")
    template_vars["valid_cert_dirs"] = []
    if ssl_dir.is_dir():
        for d in ssl_dir.iterdir():
            if not d.is_dir():
                continue

            cert_path = d / "cert.pem"
            key_path = d / "key.pem"
            domains_path = d / "domains"

            if cert_path.is_file() and key_path.is_file() and domains_path.is_file():
                domains = domains_path.read_text().strip()
                domains_list = domains.split()
                if domains_list and template_vars["SERVER_HOSTNAME"] not in domains_list:
                    template_vars["valid_cert_dirs"].append({
                        "cert_path": d.absolute().as_posix() + "/",
                        "domains": domains,
                    })

    return template_vars

def main():
    env = Environment(loader=FileSystemLoader("./etc/nginx/conf.d/templates"))

    # Render config
    print("Render config")
    template_vars = prepare_template_vars(os.environ)
    sites_default_conf(env, template_vars)
    nginx_conf(env, template_vars)
    includes_conf(env, template_vars)


if __name__ == "__main__":
    main()
