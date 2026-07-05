from pyvis.network import Network
import networkx as nx
import pandas as pd

from data_loader import load_and_build_graph as load_and_build_graph

def visualize_hetero_graph_pyvis(data, clause_to_idx, entity_to_idx=None, filename="legal_graph.html"):
    """
    HeteroData 객체를 추출하여 대화형 HTML 파일로 시각화합니다.
    """
    # 역매핑 딕셔너리 생성 (인덱스 -> 원래 텍스트 명칭)
    idx_to_clause = {v: k for k, v in clause_to_idx.items()}
    
    # 만약 entity_to_idx가 없다면 그래프 빌드 시의 set(subjects+objects) 순서대로 복원하거나
    # 임시로 "Entity_{idx}" 형태로 표현합니다. 여기서는 역매핑이 있다고 가정하거나 생성합니다.
    if entity_to_idx:
        idx_to_entity = {v: k for k, v in entity_to_idx.items()}
    else:
        idx_to_entity = {}

    # 1. NetworkX MultiDiGraph 객체 생성 (다중 유향 그래프)
    G = nx.MultiDiGraph()

    # 2. 노드 추가 및 그룹 설정
    for c_text, c_idx in clause_to_idx.items():
        G.add_node(f"C_{c_idx}", label=c_text, title=f"Type: Clause\n{c_text}", group="clause")
        
    for e_text, e_idx in entity_to_idx.items() if entity_to_idx else {}:
        G.add_node(f"E_{e_idx}", label=e_text, title=f"Type: Entity\n{e_text}", group="entity")

    # 3. 모든 엣지 타입을 순회하며 그래프에 연결 관계 추가
    for edge_type in data.edge_types:
        src_type, rel_type, dst_type = edge_type
        edge_index = data[edge_type].edge_index.numpy()
        
        # 관계 이름 소문자화
        rel_name = str(rel_type).lower()
        
        for i in range(edge_index.shape[1]):
            src_idx = edge_index[0, i]
            dst_idx = edge_index[1, i]
            
            # 노드 ID 접두사 설정 ('C_' 또는 'E_')
            src_id = f"C_{src_idx}" if src_type == "clause" else f"E_{src_idx}"
            dst_id = f"C_{dst_idx}" if dst_type == "clause" else f"E_{dst_idx}"
            
            # 존재하지 않는 Entity 노드가 있다면 동적 추가
            if not G.has_node(src_id) and src_type == "entity":
                name = idx_to_entity.get(src_idx, f"Entity_{src_idx}")
                G.add_node(src_id, label=name, title=name, group="entity")
            if not G.has_node(dst_id) and dst_type == "entity":
                name = idx_to_entity.get(dst_idx, f"Entity_{dst_idx}")
                G.add_node(dst_id, label=name, title=name, group="entity")

            # 엣지 추가 (관계 종류를 label로 지정)
            G.add_edge(src_id, dst_id, label=rel_name, title=rel_name)

    # 4. PyVis Network로 변환 및 시각화 설정
    net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black", directed=True)
    net.from_nx(G)
    
    # 물리 엔진 및 탐색 옵션 활성화
    net.show_buttons(filter_=['physics', 'selection'])
    net.save_graph(filename)
    print(f"시각화 완료! 브라우저에서 '{filename}' 파일을 열어보세요.")

# 사용 예시:
if __name__ == "__main__":
    data, c2i, e2i = load_and_build_graph('./data/nodes_20260626_171919.csv', './data/triplets_20260626_171919.csv', use_dummy_emb=False)
    visualize_hetero_graph_pyvis(data, c2i, e2i)
