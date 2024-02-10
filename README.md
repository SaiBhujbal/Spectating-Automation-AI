# Spectating-Automation-AI

## Esports Spectating Automation using AI

Esports, with its 4.3 billion dollar global economy, has become a significant industry with a massive fanbase. Spectating in esports involves streaming players' Points of View (POVs) to the audience, but with multiple players in tournaments, manually streaming all of them is impractical. This project focuses on automating the spectating process using AI.

### Factors Considered for POV Switching

1. **Euclidean Distance:**
   - Calculates the distance between enemy players.
   - Pairs with shorter distances are given higher priority.
   
2. **Edge Presence:**
   - Uses Euclidean distance along with the presence of edges between players on the map.
   - Edges represent walls and obstructions; no edges mean players can shoot through.
   - Switches POV with higher priority if there are no edges between enemy players.

### Clustering Effect

- Detects clustering events where a group of players gathers in the same area.
- Adjusts spectating behavior to switch players rapidly in clustered areas.
- Prevents missing thrilling combat situations, especially during rapid kills.
