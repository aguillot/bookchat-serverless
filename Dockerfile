FROM public.ecr.aws/lambda/python:3.11

RUN yum install -y gcc-c++

# Copy function code
COPY handler.py schemas.py ${LAMBDA_TASK_ROOT}

# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install the specified packages
RUN pip install -r requirements.txt --no-cache-dir

RUN python -m nltk.downloader -d /usr/local/share/nltk_data popular
