from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

INPUT_PATH = Path("reports/tables/business_impact_scenarios.csv")
OUTPUT_PATH = Path("reports/figures/business/business_impact_incremental.png")

def main():
    df = pd.read_csv(INPUT_PATH)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    # Usar nombres de columnas exactos de tu CSV
    # Nota de Pippi: Verifica que la columna se llame exactamente así en tu CSV
    ax.bar(df["Escenario"], df["Impacto_incremental_modelo"], color='skyblue')
    
    ax.set_title("Impacto Incremental del Modelo vs Azar (€)")
    ax.set_ylabel("Valor Económico Salvado")
    ax.set_xlabel("Escenarios")
    
    # Añadir los números encima de las barras
    for i, value in enumerate(df["Impacto_incremental_modelo"]):
        ax.text(i, value + 500, f"{value:,.0f}€", ha="center", va="bottom", fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close()
    
    print("=== Gráfico de Impacto de Negocio Generado ===")
    print(f"Guardado: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()