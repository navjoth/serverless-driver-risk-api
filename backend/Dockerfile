FROM public.ecr.aws/lambda/python:3.12

# Copy function code
COPY app.py ./
COPY model.pkl ./
COPY requirements.txt ./

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Set the CMD to your handler (function name inside app.py)
CMD ["app.lambda_handler"]
