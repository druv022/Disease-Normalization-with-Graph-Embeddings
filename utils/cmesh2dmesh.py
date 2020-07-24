# python imports
from SPARQLWrapper import JSON, SPARQLWrapper


def getMeSH(concepts_query, sparql, graph_uri, concept):
    """
    Input:  query, sparql connection, graph URI (optional)
    Output: list of (cuids, muids) pairs
        """
    sparql.setQuery(concepts_query.format(concept=concept, graph_uri=graph_uri))
    print(concepts_query.format(concept=concept, graph_uri=graph_uri))
    sparql.setReturnFormat(JSON)
    concepts_results = sparql.query().convert()

    return [str(row['muid']['value']) for row in concepts_results['results']['bindings']]


if __name__ == '__main__':


   print('building CUID to MUID mapping')

   mesh_query = """
   select distinct ?muid
   {{ graph {graph_uri} {{
          <{concept}> <http://id.nlm.nih.gov/mesh/vocab#preferredMappedTo> ?muid .
       }}
   }}
   """

   sparql      = SPARQLWrapper("https://id.nlm.nih.gov/mesh/query")
   graph_uri   = 'http://id.nlm.nih.gov/mesh'
   in_data     = 'C.txt'
   out_data    = 'C2M_mesh.txt'

   f = open(in_data,  'r')
   g = open(out_data, 'w')

   count = 0

   for line in f.readlines():
       concept = 'http://id.nlm.nih.gov/mesh/2018/' + line.strip()
       res = getMeSH(mesh_query, sparql, graph_uri, concept)
       print('\n--', concept, 'has', len(res), 'candidates')
       for d in res:
           print('\t', concept, '\t', d)
           g.write(concept + '\t' + d + '\n')
           count = count + 1

   print('\n==> found', count, 'mappings')

   f.close()
   g.close()
