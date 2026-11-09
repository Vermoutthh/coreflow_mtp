from coreflow_mtp.dag.dag import create_dag_by_json


class Query:
    def __init__(self, query_id, work_dir, node_dependency, node_list,
                 query_submit_time=0.0):
        self.query_id = query_id
        self.query_submit_time = query_submit_time
        self.dag = create_dag_by_json(work_dir, node_dependency, node_list)

    def get_entry_nodes(self):
        return self.dag.get_entry_nodes()

    def get_next_execute_agents(self, invocation_id, query_id, repeat, result=None):
        return self.dag.update_result(invocation_id, query_id, repeat, result=result)

    def is_completed(self):
        return self.dag.is_completed()

    def get_latency(self):
        if not self.is_completed():
            raise ValueError("Query is not complete")
        return self.dag.end_to_end_latency
