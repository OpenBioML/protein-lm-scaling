import csv
import gzip
from xml.etree import ElementTree as ET
import click
import argparse
import pandas as pd

DB_FILENAME = 'uniref_mapping.db'
NAMESPACES = {'uniref': 'http://uniprot.org/uniref'}
ENTRY_TAG = '{{{}}}entry'.format(NAMESPACES['uniref'])


def parse_entry(entry):
    """Returns UniRef100, UniRef90 and UniRef50 IDs for a given ``entry``"""
    uniref100_id = entry.attrib['id']

    db_ref = entry.findall(
        './/uniref:representativeMember/uniref:dbReference',
        namespaces=NAMESPACES
    )
    db_ref = db_ref[0]

    uniref90_id = db_ref.findall(
        './/uniref:property[@type="UniRef90 ID"]', namespaces=NAMESPACES)
    if uniref90_id:
        uniref90_id = uniref90_id[0].attrib['value']
    else:
        uniref90_id = ''

    uniref50_id = db_ref.findall(
        './/uniref:property[@type="UniRef50 ID"]', namespaces=NAMESPACES)
    if uniref50_id:
        uniref50_id = uniref50_id[0].attrib['value']
    else:
        uniref50_id = ''
    
    representative_member = entry.findall(
        './/uniref:representativeMember/uniref:sequence',
        namespaces=NAMESPACES
    )[0].text

    cluster_size = entry.findall(
        './/uniref:property[@type="member count"]',
        namespaces=NAMESPACES
    )
    if cluster_size:
        cluster_size = cluster_size[0].attrib['value']
    else:
        cluster_size = 0
    
    return uniref100_id, uniref90_id, uniref50_id, representative_member, cluster_size

def make_uniref_mapping(uniref_xml_db, output_file, num_total_sequences):
    row_count = 0
    progress_bar_batch_size = 10000
    progress_bar_kwargs = {
        'length': num_total_sequences, 
    }
    click.echo('Writing mapping file to {}'.format(output_file))

    with click.progressbar(**progress_bar_kwargs) as bar, \
            gzip.open(uniref_xml_db) as f_in, \
            open(output_file, 'wt') as f_out:

        parser = ET.iterparse(f_in, events=('start', 'end'))
        _, root = next(parser)
        #mytree = ET.parse(uniref_xml_db)
        #root = mytree.getroot()
        output_writer = csv.writer(f_out, delimiter=',')

        # Write the header for the output file
        output_writer.writerow(['UniRef100', 'UniRef90', 'UniRef50', 'Representative_member_Uniref100', 'cluster_size_U100'])

        for event, element in parser:
            if element.tag == ENTRY_TAG and event == 'end':
                row = parse_entry(element)

                if len(row) < 3:
                    el = row[0] if len(row) > 0 else 'N/A'
                    click.secho(
                        'Element "{}" does not have mappings!'.format(el),
                        color='yellow',
                    )
                    continue

                row_count += 1
                output_writer.writerow(row)

                if row_count % progress_bar_batch_size == 0:
                    bar.update(progress_bar_batch_size)

                # Free up memory by removing the entry from the tree
                element.clear()
                root.remove(element)

        bar.update(row_count % progress_bar_batch_size)

    click.echo('Done. {} sequences mapped.'.format(row_count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess fasta data')
    parser.add_argument('--uniref100_xml_path', default=None, type=str, help='Path to raw fasta data')
    parser.add_argument('--output_file', default=None, type=str, help='Path to processed training/validation data')
    parser.add_argument('--num_total_sequences', default=None, type=int, help='Num of Uniref100 clusters')
    parser.add_argument('--cluster_size_U90_file', default=None, type=str, help='File with size for all U90 clusters')
    parser.add_argument('--cluster_size_U50_file', default=None, type=str, help='File with size for all U90 clusters')
    args = parser.parse_args()

    make_uniref_mapping(args.uniref100_xml_path, args.output_file, args.num_total_sequences)
    final = pd.read_csv(args.output_file)
    u90_cluster_sizes=pd.read_csv(args.cluster_size_U90_file)
    u50_cluster_sizes=pd.read_csv(args.cluster_size_U50_file)[['cluster_ID','cluster_size']]
    final=pd.merge(final,u90_cluster_sizes,how='left',left_on='UniRef90',right_on='cluster_ID')
    final.rename(columns={'cluster_size':'cluster_size_U90'}, inplace=True)
    final=pd.merge(final,u50_cluster_sizes,how='left',left_on='UniRef50',right_on='cluster_ID')
    final.rename(columns={'cluster_size':'cluster_size_U50'}, inplace=True)
    final.to_csv(args.output_file.split('.csv')[0]+'_final.csv', index=False)
    print(final.head())