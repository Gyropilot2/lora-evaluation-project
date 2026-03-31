import { useEffect, useRef, useState } from "react";
import { TrendPlot, trendGridTicks, trendPointX, type TrendCeilings, type TrendPoint, type TrendScale } from "./StrengthTrend";

export function TrendChartFrame({ points, scale, ceilings }: { points: TrendPoint[]; scale: TrendScale; ceilings: TrendCeilings }) {
  const minX = Math.min(...points.map((point) => point.value));
  const maxX = Math.max(...points.map((point) => point.value));
  const xRange = maxX - minX;
  const plotRef = useRef<HTMLDivElement | null>(null);
  const [plotSize, setPlotSize] = useState({ width: 96, height: 96 });

  useEffect(() => {
    const element = plotRef.current;
    if (!element) return;

    const updateSize = () => {
      const rect = element.getBoundingClientRect();
      const width = Math.max(32, Math.round(rect.width));
      const height = Math.max(32, Math.round(rect.height));
      setPlotSize((current) => (current.width === width && current.height === height ? current : { width, height }));
    };

    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const middleTicks = trendGridTicks(scale.min, ceilings.badness);

  return (
    <div className="trend-sparkline">
      <div className="trend-chart-bundle">
        <div className="trend-plot-column" ref={plotRef}>
          <TrendPlot points={points} scale={scale} ceilings={ceilings} width={plotSize.width} height={plotSize.height} yGridTicks={middleTicks} />
        </div>
        <div className="trend-labels">
          {points.map((point, index) => {
            const x = trendPointX(point.value, minX, xRange, plotSize.width);
            const transform =
              xRange === 0 ? "translateX(-50%)" : index === 0 ? "translateX(0)" : index === points.length - 1 ? "translateX(-100%)" : "translateX(-50%)";
            return (
              <span
                key={`${point.label}-${index}`}
                className="trend-label"
                style={{ left: `${x}px`, transform }}
              >
                {point.value.toFixed(1)}
              </span>
            );
          })}
        </div>
      </div>
    </div>
  );
}
