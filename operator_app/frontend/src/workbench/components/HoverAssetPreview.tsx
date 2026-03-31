import "./HoverAssetPreview.css";

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";

import { reviewAssetPreviewUrl } from "../reviewClient";

type HoverAssetPreviewProps = {
  assetPath: string | null | undefined;
  label: string;
  children: ReactNode;
  className?: string;
  primaryLabel?: string | null;
  secondaryAssetPath?: string | null | undefined;
  secondaryLabel?: string | null;
};

type PopupPosition = {
  left: number;
  top: number;
};

const POPUP_GAP = 10;

export function HoverAssetPreview({
  assetPath,
  label,
  children,
  className = "",
  primaryLabel,
  secondaryAssetPath,
  secondaryLabel,
}: HoverAssetPreviewProps) {
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const [visible, setVisible] = useState(false);
  const [position, setPosition] = useState<PopupPosition>({ left: 0, top: 0 });
  const previewUrl = useMemo(() => reviewAssetPreviewUrl(assetPath), [assetPath]);
  const secondaryPreviewUrl = useMemo(() => reviewAssetPreviewUrl(secondaryAssetPath), [secondaryAssetPath]);
  const popupWidth = secondaryPreviewUrl ? 520 : 256;

  function updatePosition() {
    const trigger = triggerRef.current;
    if (!trigger) return;
    const rect = trigger.getBoundingClientRect();
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    const preferredLeft = rect.left - popupWidth - POPUP_GAP;
    const fallbackLeft = rect.right + POPUP_GAP;
    const left =
      preferredLeft >= 12
        ? preferredLeft
        : Math.min(fallbackLeft, Math.max(12, viewportWidth - popupWidth - 12));
    const top = Math.min(Math.max(12, rect.top - 8), Math.max(12, viewportHeight - 320));
    setPosition({ left, top });
  }

  function showPopup() {
    if (!previewUrl) return;
    updatePosition();
    setVisible(true);
  }

  function hidePopup() {
    setVisible(false);
  }

  useEffect(() => {
    if (!visible) return undefined;
    const onViewportChange = () => updatePosition();
    window.addEventListener("resize", onViewportChange);
    window.addEventListener("scroll", onViewportChange, true);
    return () => {
      window.removeEventListener("resize", onViewportChange);
      window.removeEventListener("scroll", onViewportChange, true);
    };
  }, [popupWidth, visible]);

  const triggerClassName = `hover-preview-trigger${className ? ` ${className}` : ""}`;

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        className={triggerClassName}
        onMouseEnter={showPopup}
        onMouseLeave={hidePopup}
        onFocus={showPopup}
        onBlur={hidePopup}
        disabled={!previewUrl}
      >
        {children}
      </button>
      {visible && previewUrl && typeof document !== "undefined"
        ? createPortal(
            <div
              className="hover-preview-popup"
              style={{ ...position, width: popupWidth }}
              onMouseEnter={showPopup}
              onMouseLeave={hidePopup}
            >
              <div className="hover-preview-popup-header">{label}</div>
              <div className={`hover-preview-gallery${secondaryPreviewUrl ? " is-paired" : ""}`}>
                {secondaryPreviewUrl ? (
                  <div className="hover-preview-pane">
                    <div className="hover-preview-pane-label">{secondaryLabel ?? "Comparison"}</div>
                    <img
                      className="hover-preview-image"
                      src={secondaryPreviewUrl}
                      alt={secondaryLabel ?? `${label} comparison`}
                    />
                  </div>
                ) : null}
                <div className="hover-preview-pane">
                  <div className="hover-preview-pane-label">
                    {primaryLabel ?? (secondaryPreviewUrl ? "Focused" : "Preview")}
                  </div>
                  <img className="hover-preview-image" src={previewUrl} alt={label} />
                </div>
              </div>
            </div>,
            document.body,
          )
        : null}
    </>
  );
}
