# Use the ultra-lightweight Nginx alpine image
FROM nginx:1.25-alpine

# Copy raw HTML, CSS, and JS files directly to Nginx public folder
COPY index.html /usr/share/nginx/html/
COPY style.css /usr/share/nginx/html/
COPY app.js /usr/share/nginx/html/

# Copy custom Nginx configuration for routing and reverse proxy
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
