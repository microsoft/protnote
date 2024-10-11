import requests
import argparse

def upload(filename, path, bucket, access_token):
  # This is verbatim from the docs, but they forgot the params section
  with open(path, "rb") as fp:
      r = requests.put(
          "%s/%s" % (bucket, filename),
          data=fp,
          params={'access_token': access_token},
      )
  return r.json()


# Main script using argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a file to Zenodo")
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="The file to upload",
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="The path to the file",
    )
    #make deposition id and access token part of arguments
    parser.add_argument(
        "--deposition-id",
        type=int,
        required=True,
        help="The deposition id",
    )
    parser.add_argument(
        "--access-token",
        type=str,
        required=True,
        help="The access token",
    )

    args = parser.parse_args()

    r = requests.get(f'https://zenodo.org/api/deposit/depositions/{args.deposition_id}',
                    params={'access_token': args.access_token})
    r.status_code
    data = r.json()

    response = upload(args.file, args.path, data['links']['bucket'], args.access_token)
    print(response)