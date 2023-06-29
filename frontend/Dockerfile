FROM node:16-alpine

# set working directory
WORKDIR /usr/src/fer-front
COPY . .

# install and cache app dependencies
COPY package.json .
COPY yarn.lock .
RUN yarn install

# start app
CMD ["yarn", "start"]