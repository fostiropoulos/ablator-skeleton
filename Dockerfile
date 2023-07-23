FROM ablator
WORKDIR /usr/src/app
LABEL maintainer="you@example.com"
LABEL description="Building AGI"
COPY . .
RUN pip install .
CMD ["python","-m","ablator_skeleton","--mp","mock_param"]
