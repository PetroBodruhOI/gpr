import { useEffect, useState } from "react";

// Three pages, addressed by hash so the static Vercel deploy needs no rewrites:
//   #/  home   ·   #/patterns   ·   #/detect
export type Route = "home" | "patterns" | "detect";

export const href = (r: Route) => (r === "home" ? "#/" : `#/${r}`);

function parse(hash: string): Route {
  const path = hash.replace(/^#\/?/, "");
  return path === "patterns" || path === "detect" ? path : "home";
}

export function useHashRoute(): [Route, (r: Route) => void] {
  const [route, setRoute] = useState<Route>(() => parse(window.location.hash));

  useEffect(() => {
    const onChange = () => {
      setRoute(parse(window.location.hash));
      window.scrollTo({ top: 0 });
    };
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);

  const navigate = (r: Route) => {
    window.location.hash = href(r);
  };
  return [route, navigate];
}

// Adds data-inview to the element once it scrolls into view (for .reveal).
export function useReveal<T extends HTMLElement>() {
  const [el, setEl] = useState<T | null>(null);
  useEffect(() => {
    if (!el) return;
    if (!("IntersectionObserver" in window)) {
      el.setAttribute("data-inview", "");
      return;
    }
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          el.setAttribute("data-inview", "");
          io.disconnect();
        }
      },
      { rootMargin: "0px 0px -15% 0px" },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [el]);
  return setEl;
}
