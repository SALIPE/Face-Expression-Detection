FROM node:18-alpine

# set working directory
WORKDIR /usr/src/fer
COPY . .


# install and cache app dependencies
COPY package.json .
COPY yarn.lock .
RUN yarn install

EXPOSE 3000
# start app
CMD ["yarn", "start"]