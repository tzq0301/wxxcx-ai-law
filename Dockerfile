# start by pulling the python image
FROM python:3.9-alpine

# copy the requirements file into the image
COPY ./requirements.txt /app/requirements.txt

# switch working directory
WORKDIR /app

RUN python3 -m pip install --upgrade pip
RUN pip3 install scipy
RUN pip3 install jieba
RUN pip3 install numpy
RUN pip3 install flask
RUN pip3 install gensim

# install the dependencies and packages in the requirements file
#RUN pip install -r requirements.txt

# copy every content from the local file to the image
COPY . /app

# configure the container to run in an executed manner
ENTRYPOINT [ "python" ]

CMD [ "app.py" ]